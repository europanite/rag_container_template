import React from "react";
import { View, Text, TouchableOpacity, useWindowDimensions, Linking } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { StatusBar } from "expo-status-bar";
import { useAuth } from "../context/Auth";
import { useNavigation } from "@react-navigation/native";

const BAR_BG = "#000000ff";
const CONTENT_MAX_W = 720;

export default function SettingsBar() {
  const { user, signOut } = useAuth();
  const nav = useNavigation<any>();
  const { width } = useWindowDimensions();
  const isNarrow = width < 420;
  const NOT_SIGNED_COLOR = BAR_BG;

  const Btn = ({ title, onPress }: { title: string; onPress: () => void }) => (
    <TouchableOpacity
      onPress={onPress}
      style={{
        backgroundColor: "#fff",
        paddingVertical: 8,
        paddingHorizontal: 12,
        borderRadius: 6,
        borderWidth: 1,
      }}
    >
      <Text style={{ fontWeight: "600" }}>{title}</Text>
    </TouchableOpacity>
  );

  const REPO_URL = "https://github.com/europanite/rag_container_template";

  return (
    <SafeAreaView edges={["top"]} style={{ backgroundColor: BAR_BG }}>
      <StatusBar style="dark" backgroundColor={BAR_BG} />
      <View style={{ backgroundColor: BAR_BG, paddingHorizontal: 12, paddingVertical: 10 }}>
        <View style={{ alignItems: "center" }}>
          <View style={{ width: "100%", maxWidth: CONTENT_MAX_W, gap: 8 }}>
            <View
              style={{
                flexDirection: isNarrow ? "column" : "row",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 8,
                flexWrap: "wrap",
              }}
            >
              <View style={{ flexShrink: 1, gap: 2 }}>
                <Text
                  style={{
                    color: "#e5e7eb",
                    fontSize: 32,
                    textDecorationLine: "underline",
                  }}
                  onPress={() => Linking.openURL(REPO_URL)}
                >
                  RAG Container Template
                </Text>
              </View>

              <View style={{ flexDirection: "row", gap: 8, flexWrap: "wrap" }}>
                <Text style={{ fontWeight: "700", color: "#ffffffff" }}>
                  {user ? `${user.email}` : "Not signed in"}
                </Text>
                {!user ? (
                  <>
                    <Btn title="Sign Up" onPress={() => nav.navigate("SignUp")} />
                    <Btn title="Sign In" onPress={() => nav.navigate("SignIn")} />
                  </>
                ) : (
                  <Btn title="Sign out" onPress={signOut} />
                )}
              </View>
            </View>
          </View>
        </View>
      </View>
    </SafeAreaView>
  );
}
