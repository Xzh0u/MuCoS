import javalang
from utils import *
import json


def get_token_list(tokens):
    record_tokens = []
    start_record_token = 0

    # only count tokens in method body
    for token in tokens:
        if token.value == '{' and start_record_token < 2:
            start_record_token += 1
        if start_record_token >= 2 and type(token) == javalang.tokenizer.Identifier:
            record_tokens.append(token.value)

    return record_tokens


def parse_tokens(record_tokens):
    splited_tokens = []
    for record_token in record_tokens:
        if '_' in record_token:
            split_result = split_(record_token)
        else:
            split_result = camel_case_split(record_token)
        splited_tokens.extend(x.lower() for x in split_result)

    splited_tokens = list(set(splited_tokens))  # remove duplicate

    return remove_stop_and_key_words(splited_tokens)


def extract_method_name(code_snap) -> List:
    try:
        tokens = list(javalang.tokenizer.tokenize(code_snap))
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
    except:
        print('Syntax Error(extract name)')
        return []

    method_name = tree.body[0].name
    splited_names = camel_case_split(method_name)

    return [x.lower() for x in splited_names]


def extract_tokens(code_snap):
    javalang_tokens = list(javalang.tokenizer.tokenize(code_snap))

    record_tokens = get_token_list(javalang_tokens)
    parsed_tokens = parse_tokens(record_tokens)

    # remove the word that the length is 1
    parsed_tokens = remove_short_words(2, parsed_tokens)
    return parsed_tokens


def extract_api(code_snap):
    apis = []
    try:
        tree = javalang.parse.parse(code_snap)
    except:
        print('Syntax Error(extract api)')
        return []

    with open('../data/vocab.apiseq.json') as f:
        vocab = json.load(f)

    # extract method invocation
    for path, node in tree.filter(javalang.tree.MethodInvocation):
        if node.member in vocab and node.qualifier != None and node.member != None:
            if node.qualifier != "" and node.member != "":
                apis.extend([node.qualifier, node.member])

    # extract constructor invocation
    for path, node in tree.filter(javalang.tree.ClassCreator):
        apis.extend([node.type.name, "new"])

    return apis


def main():
    filename = "/home/v-xiaoshi/dataset/data/test_code_snaps.txt"
    for code_snap in read_tsv(filename):
        standard_code_snap = "public class Example {" + code_snap + "}"

        print("Processed API: ", extract_api(standard_code_snap))
        print("Tokens: ", extract_tokens(standard_code_snap))
        print("Method Names: ", extract_method_name(standard_code_snap))

    # Test code snaps:
    # code_snap = "public class Example { \n public static Calendar toCalendar(final Date date) { \n final Calendar c = Calendar.getInstance();\n c.setTime(date);\n return c;\n }\n }"
    # code_snap2 = "public class Example { public RegistryContext ( String host , int port , Hashtable < ? , ? > env ) throws NamingException { environment = ( env == null ) ? new Hashtable < String , Object > ( 5 ) : ( Hashtable < String , Object > ) env ; if ( environment . get ( SECURITY_MGR ) != null ) { installSecurityMgr ( ) ; } if ( ( host != null ) && ( host . charAt ( 0 ) == '[' ) ) { host = host . substring ( 1 , host . length ( ) - 1 ) ; } RMIClientSocketFactory socketFactory = ( RMIClientSocketFactory ) environment . get ( SOCKET_FACTORY ) ; registry = getRegistry ( host , port , socketFactory ) ; this . host = host ; this . port = port ; } }"
    # code_snap3 = "public class Example { public int singleNumberWithMap ( int [ ] nums ) { Map < Integer , Integer > map = new HashMap < Integer , Integer > ( ) ; for ( int i : nums ) { if ( map . containsKey ( i ) ) { map . put ( i , map . get ( i ) + 1 ) ; } else { map . put ( i , 1 ) ; } } for ( Map . Entry < Integer , Integer > entry : map . entrySet ( ) ) { if ( entry . getValue ( ) < 3 ) { return entry . getKey ( ) ; } } for ( Map . Entry < Integer , Integer > entry : map . entrySet ( ) ) { if ( entry . getValue ( ) < 3 ) { return entry . getKey ( ) ; } } for ( Object o : map . entrySet ( ) ) { Map . Entry entry = ( Map . Entry ) o ; if ( ( Integer ) entry . getValue ( ) == 1 ) { return ( Integer ) entry . getKey ( ) ; } } return 0 ; } }"


if __name__ == "__main__":
    main()